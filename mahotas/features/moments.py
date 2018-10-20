# Copyright (C) 2008-2018, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division
import numpy as np

__all__ = ['moments']
def moments(img, p0, p1, cm=None, convert_to_float=True, normalize=False, normalise=False):
    '''
    m = moments(img, p0, p1, cm=(0, 0), convert_to_float=True)

    Returns the p0-p1 moment of image `img`

    The formula computed is

    \sum_{ij} { img[i,j] (i - c0)**p0 (j - c1)**p1 }

    where cm = (c0,c1). If `cm` is not given, then (0,0) is used.

    If image is of an integer type, then it is internally converted to
    np.float64, unlesss `convert_to_float` is False. The reason is that,
    otherwise, overflow is likely except for small images. Since this
    conversion takes longer than the computation, you can turn it off in case
    you are sure that your images are small enough for overflow to be an issue.
    Note that no conversion is made if `img` is of any floating point type.

    Parameters
    ----------
    img : 2-ndarray
        An 2-d ndarray
    p0 : float
        Power for first dimension
    p1 : float
        Power for second dimension
    cm : (int,int), optional
        center of mass (default: 0,0)
    convert_to_float : boolean, optional
        whether to convert to floating point (default: True)
    normalize : boolean, optional
        whether to normalize to size of image (default: False)

    Returns
    -------
    moment: float
        floating point number

    Notes
    -----
    It only works for 2-D images
    '''
    if normalise:
        normalize = True
    if not np.issubdtype(img.dtype, np.floating) and convert_to_float:
        img = img.astype(np.float64)
    r,c = img.shape
    p = np.arange(c, dtype=float)
    if cm is not None:
        p -= cm[1]
    p **= p1
    if normalize:
        p /= p.sum()
    inter = np.dot(img, p)
    p = np.arange(r, dtype=float)
    if cm is not None:
        p -= cm[0]
    p **= p0
    if normalize:
        p /= p.sum()
    return np.dot(inter, p)

