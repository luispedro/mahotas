# -*- coding: utf-8 -*-
# Copyright (C) 2008-2026, Luis Pedro Coelho <luis@luispedro.org>
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

The global thresholding functions (``otsu`` and ``rc``) return **the numeric
threshold**. In order to obtain a *thresholded image*, you can do the
following::

    T_otsu = mh.otsu(image)
    binarized_image = (image > T_otsu)

Functions which have an ``ignore_zeros`` parameters will only consider non-zero
pixels when computing the thresholding.

The local thresholding functions (``bernsen`` and ``gbernsen``) compute a
different threshold for each pixel and return the *thresholded (binary)
image* directly. ``soft_threshold`` returns a transformed (non-binary) image.

Global thresholding (return a threshold value):

- ``otsu()``: Otsu method
- ``rc()``: Riddler-Calvard method

Local thresholding (return a binary image):

- ``bernsen()``: Bernsen thresholding (circular neighbourhood)
- ``gbernsen()``: Generalized Bernsen thresholding (arbitrary neighbourhood)

Other:

- ``soft_threshold()``: soft thresholding (shrinkage towards zero)

Only ``otsu`` and ``rc`` are re-exported at the top-level ``mahotas``
namespace; the others must be accessed as ``mahotas.thresholding.<name>``.
'''

import numpy as np
from numpy.typing import NDArray

from ._typing import Array
from .histogram import fullhistogram
from . import _histogram
from .internal import _verify_is_integer_type
__all__ = [
        'otsu',
        'rc',
        'soft_threshold',
        'bernsen',
        'gbernsen',
    ]


def otsu(img: Array, ignore_zeros: bool = False) -> int:
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
    img : ndarray
        Image of an unsigned integer type.
    ignore_zeros : boolean, optional
        whether to ignore zero-valued pixels
        (default: False)

    Returns
    -------
    T : integer
        the threshold

    Examples
    --------
    >>> import numpy as np
    >>> import mahotas as mh
    >>> im = np.array([[10, 12, 11], [200, 210, 205]], np.uint8)
    >>> t = mh.otsu(im)
    >>> t
    12
    >>> im > t
    array([[False, False, False],
           [ True,  True,  True]])
    """
    _verify_is_integer_type(img, 'otsu')
    hist = fullhistogram(img)
    hist = np.asanyarray(hist, dtype=np.double)
    if ignore_zeros:
        hist[0] = 0
    return _histogram.otsu(hist)


def rc(img: Array, ignore_zeros: bool = False) -> int | np.float64:
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
        Image of an unsigned integer type (as for ``otsu``)
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

def soft_threshold(f: Array, tval: float) -> Array:
    '''
    thresholded = soft_threshold(f, tval)

    Soft threshold function (shrinkage towards zero)

    Values with absolute value at most ``tval`` are set to 0, while others are
    moved ``tval`` closer to 0, i.e., ``sign(f) * max(abs(f) - tval, 0)``::

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
        input array
    tval : scalar
        threshold (non-negative)

    Returns
    -------
    thresholded : ndarray
        Array of the same shape as `f`
    '''

    f = f * (np.abs(f) > tval)
    f -= tval * (f > tval)
    f += tval * (f < -tval)
    return f

def bernsen(
    f: Array, radius: int, contrast_threshold: float, gthresh: float | None = None,
) -> NDArray[np.bool_]:
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
        global threshold to fall back in low contrast regions (default: 128,
        which assumes an 8-bit image).
        Note that, unlike here, this argument is *required* in ``gbernsen``.

    Returns
    -------
    thresholded : binary ndarray

    See Also
    --------
    gbernsen : function
        Generalised Bernsen thresholding. ``bernsen(f, r, c, g)`` is
        equivalent to ``gbernsen(f, mahotas.morph.circle_se(r), c, g)``.
    '''
    from mahotas.morph import circle_se
    if gthresh is None:
        gthresh = 128
    return gbernsen(f, circle_se(radius), contrast_threshold, gthresh)

def gbernsen(f: Array, se: Array, contrast_threshold: float, gthresh: float) -> NDArray[np.bool_]:
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
    gthresh : numeric
        global threshold to fall back in low contrast regions. Unlike in
        ``bernsen`` (where it defaults to 128), this argument is required.

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
    return np.choose(fptp < contrast_threshold, (f > fmean, f > gthresh))
