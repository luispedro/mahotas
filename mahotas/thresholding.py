# -*- coding: utf-8 -*-
# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
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
Thresholding Module
===================

Thresholding functions:

:otsu(): Otsu method
:rc(): Riddler-Calvard's method
'''

from __future__ import division
import numpy as np
from .histogram import fullhistogram
__all__ = [
        'otsu',
        'rc',
    ]


def otsu(img, ignore_zeros=False):
    """
    T = otsu(img, ignore_zeros=False)

    Calculate a threshold according to the Otsu method.

    Parameters
    ----------
    img : an image as a numpy array.
        This should be of an unsigned integer type.
    ignore_zeros : Boolean
        whether to ignore zero-valued pixels
        (default: False)

    Returns
    -------
    T : integer
        the threshold
    """
# Calculated according to CVonline:
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/threshold.pdf
    hist = fullhistogram(img)
    hist = hist.astype(np.double)
    if ignore_zeros:
        hist[0] = 0
    Ng = len(hist)
    nB = np.cumsum(hist)
    nO = nB[-1]-nB
    mu_B = 0
    mu_O = (np.arange(1, Ng)*hist[1:]).sum()/hist[1:].sum()
    best = nB[0]*nO[0]*(mu_B-mu_O)*(mu_B-mu_O)
    bestT = 0

    for T in xrange(1, Ng):
        if nB[T] == 0: continue
        if nO[T] == 0: break
        mu_B = (mu_B*nB[T-1] + T*hist[T]) / nB[T]
        mu_O = (mu_O*nO[T-1] - T*hist[T]) / nO[T]
        sigma_between = nB[T]*nO[T]*(mu_B-mu_O)*(mu_B-mu_O)
        if sigma_between > best:
            best = sigma_between
            bestT = T
    return bestT


def rc(img, ignore_zeros=False):
    """
    T = rc(img, ignore_zeros=False)

    Calculate a threshold according to the Riddler-Calvard method.

    Parameters
    ----------
    img : ndarray
        Image of any type
    ignore_zeros : boolean, optional
        Whether to ignore zero valued pixels (default: False)

    Returns
    -------
    T : float
        threshold
    """
    hist = fullhistogram(img)
    if ignore_zeros:
        if hist[0] == img.size:
            return 0
        hist[0] = 0
    N = hist.size

    # Precompute most of what we need:
    sum1 = np.cumsum(np.arange(N) * hist)
    sum2 = np.cumsum(hist)
    sum3 = np.flipud(np.cumsum(np.flipud(np.arange(N) * hist)))
    sum4 = np.flipud(np.cumsum(np.flipud(hist)))

    maxt = N-1
    while hist[maxt] == 0:
        maxt -= 1

    res = maxt
    t = 0
    while t < min(maxt, res):
        if sum2[t] and sum4[t+1]:
            res = (sum1[t]/sum2[t] + sum3[t+1]/sum4[t+1])/2
        t += 1
    return res

