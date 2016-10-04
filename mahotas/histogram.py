# -*- coding: utf-8 -*-
# Copyright (C) 2009-2010, Luis Pedro Coelho <luis@luispedro.org>
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

'''
Histogram

:fullhistogram():
    Compute the full histogram for an image.


'''

from __future__ import division
import numpy as np
from . import _histogram
from .internal import _verify_is_integer_type

__all__ = ['fullhistogram']

def fullhistogram(img):
    """
    hist = fullhistogram(img)

    Return a histogram with bins *0, 1, ..., ``img.max()``*.

    After calling this function, it will be true that
    ``hist[i] == (img == i).sum()``, for all ``i``.

    Notes
    -----
    Only handles unsigned integer arrays.

    Parameters
    ----------
    img : array-like of an unsigned type
        input image.

    Returns
    -------
    hist : an dnarray of type np.uint32
        This will be of size ``img.max() + 1``.
    """
    _verify_is_integer_type(img, 'fullhistogram')
    img = np.ascontiguousarray(img)
    if img.dtype == np.bool:
        ones = img.sum()
        zeros = img.size - ones
        return np.array([zeros, ones], np.uintc)

    histogram = np.zeros(int(img.max()) + 1, np.uintc)
    _histogram.histogram(img, histogram)
    return histogram


