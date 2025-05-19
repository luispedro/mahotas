# -*- coding: utf-8 -*-
# Copyright (C) 2008-2019, Luis Pedro Coelho <luis@luispedro.org>
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

Thresholding functions.

These functions return **the numeric threshold**. In order to obtain a
*thresholded image*, you can do the following::

    T_otsu = mh.otsu(image)
    binarized_image = (image > T_otsu)

Functions which have an ``ignore_zeros`` parameters will only consider non-zero
pixels when computing the thresholding.

:otsu(): Otsu method
:rc(): Riddler-Calvard's method
:bernsen: Bernsen thresholding
:gbernsen: Generalized Bernsen thresholding
'''

from __future__ import division
import numpy as np
from .histogram import fullhistogram
from . import _histogram
from .internal import _verify_is_integer_type
__all__ = [
        'otsu',
        'rc',
        'soft_threshold',
        'bernsen',
        'gbernsen',,
        'elen',
    ]


def otsu(img, ignore_zeros=False):
    """
    T = otsu(img, ignore_zeros=False)

    Calculate a threshold according to the Otsu method.

    Example::

        import mahotas as mh
        import mahotas.demos

        im = mahotas.demos.nuclear_image()
        # im is stored as RGB, let's convert to single 2D format:
        im = im.max(2)

        #Now, we compute Otsu:
        t = mh.otsu(im)

        # finally, we use the value to form a binary image:
        bin = (im > t)

    See Wikipedia for details on methods:
    https://en.wikipedia.org/wiki/Otsu's_method

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
    _verify_is_integer_type(img, 'otsu')
    hist = fullhistogram(img)
    hist = np.asanyarray(hist, dtype=np.double)
    if ignore_zeros:
        hist[0] = 0
    return _histogram.otsu(hist)


def rc(img, ignore_zeros=False):
    """
    T = rc(img, ignore_zeros=False)

    Calculate a threshold according to the Riddler-Calvard method.

    Example::

        import mahotas as mh
        import mahotas.demos

        im = mahotas.demos.nuclear_image()
        # im is stored as RGB, let's convert to single 2D format:
        im = im.max(2)

        #Now, we compute a threshold:
        t = mh.rc(im)

        # finally, we use the value to form a binary image:
        bin = (im > t)

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
    first_moment = np.cumsum(np.arange(N) * hist)
    cumsum = np.cumsum(hist)

    r_first_moment = np.flipud(np.cumsum(np.flipud(np.arange(N) * hist)))
    r_cumsum = np.flipud(np.cumsum(np.flipud(hist)))

    maxt = N-1
    while hist[maxt] == 0:
        maxt -= 1

    res = maxt
    t = 0
    while t < min(maxt, res):
        if cumsum[t] and r_cumsum[t+1]:
            res = (first_moment[t]/cumsum[t] + r_first_moment[t+1]/r_cumsum[t+1])/2
        t += 1
    return res

def soft_threshold(f, tval):
    '''
    thresholded = soft_threshold(f, tval)

    Soft threshold function::

                             ^
                             |           /
                             |          /
                             |         /
                             |        /
                             |       /
         - - - - - - - - - - - - - - - - - ->
                      /      |
                     /       |
                    /        |
                   /         |
                  /          |
                 /           |

    Parameters
    ----------
    f : ndarray
    tval : scalar

    Returns
    -------
    thresholded : ndarray
    '''

    f = f * (np.abs(f) > tval)
    f -= tval * (f > tval)
    f += tval * (f < -tval)
    return f

def bernsen(f, radius, contrast_threshold, gthresh=None):
    '''
    thresholded = bernsen(f, radius, contrast_threshold, gthresh={128})

    Bernsen local thresholding

    Parameters
    ----------
    f : ndarray
        input image
    radius : integer
        radius of circle (to consider "local")
    contrast_threshold : integer
        contrast threshold
    gthresh : numeric, optional
        global threshold to fall back in low contrast regions

    Returns
    -------
    thresholded : binary ndarray

    See Also
    --------
    gbernsen : function
        Generalised Bernsen thresholding
    '''
    from mahotas.morph import circle_se
    if gthresh is None:
        gthresh = 128
    return gbernsen(f, circle_se(radius), contrast_threshold, gthresh)

def gbernsen(f, se, contrast_threshold, gthresh):
    '''
    thresholded = gbernsen(f, se, contrast_threshold, gthresh)

    Generalised Bernsen local thresholding

    Parameters
    ----------
    f : ndarray
        input image
    se : boolean ndarray
        structuring element to use for "locality"
    contrast_threshold : integer
        contrast threshold
    gthresh : numeric, optional
        global threshold to fall back in low contrast regions

    Returns
    -------
    thresholded : binary ndarray

    See Also
    --------
    bernsen : function
        Bernsen thresholding with a circular region
    '''
    from mahotas.convolve import rank_filter
    fmax = rank_filter(f, se, se.sum()-1)
    fmin = rank_filter(f, se, 0)
    fptp = fmax - fmin
    fmean = fmax/2. + fmin/2. # Do not use (fmax + fmin) as that may overflow
    return np.choose(fptp < contrast_threshold, (fmean < gthresh, fmean > f))

def elen(img, ignore_zeros=False):
    """
    T = elen(img, ignore_zeros=False)

    Calculate a threshold using Elen & Donmez's histogram-based method.

    Parameters
    ----------
    img : ndarray
        Grayscale input image (integer type preferred).
    ignore_zeros : bool, optional
        If True, ignores zero-valued pixels. Default is False.

    Returns
    -------
    T : float
        Threshold value.

    References
    ----------
    Elen, A. & Donmez, E. (2024), "Histogram-based thresholding", Optik, 306: 1-20,
    DOI: 10.1109/83.366472
    """
    _verify_is_integer_type(img, 'elen')
    hist = fullhistogram(img)
    hist = np.asanyarray(hist, dtype=np.float64)
    if ignore_zeros:
        hist[0] = 0

    total = hist.sum()
    if total == 0:
        return 0

    # Bin centers: 0, 1, ..., len(hist)-1
    bin_centers = np.arange(len(hist), dtype=np.float64)

    # Normalize histogram to obtain probability distribution
    prob = hist / total

    # Calculate global mean and standard deviation of the intensity distribution
    mean = np.sum(bin_centers * prob)
    std = np.sqrt(np.sum(((bin_centers - mean) ** 2) * prob))

    # Define alpha region as [mean - std, mean + std]
    mask_alpha = (bin_centers >= mean - std) & (bin_centers <= mean + std)

    # Separate histogram bins into alpha (central) and beta (peripheral) regions
    counts_alpha = hist[mask_alpha]
    bins_alpha = bin_centers[mask_alpha]
    counts_beta = hist[~mask_alpha]
    bins_beta = bin_centers[~mask_alpha]

    # Compute weights (sum of counts) for each region
    weight_alpha = counts_alpha.sum()
    weight_beta = counts_beta.sum()

    # Compute average intensity within each region
    avg_alpha = np.sum(bins_alpha * counts_alpha) / weight_alpha if weight_alpha > 0 else 0
    avg_beta = np.sum(bins_beta * counts_beta) / weight_beta if weight_beta > 0 else 0

    # Final threshold is the midpoint between regional means
    return (avg_alpha + avg_beta) / 2.0
