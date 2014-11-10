# Copyright (C) 2008-2014, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# License: MIT (see COPYING file)

from __future__ import division

from . import _bbox
import numpy as np

def bbox(img, border=None, as_slice=False):
    """
    min1,max1,min2,max2 = bbox(img, border={0}, as_slice={False})

    Calculate the bounding box of image img.

    Parameters
    ----------
    img : ndarray
        Any integer image type

    Returns
    -------
    min1,max1,min2,max2 : int,int,int,int
        These are such that ``img[min1:max1, min2:max2]`` contains all non-zero
        pixels. Returned when ``as_slice`` is false (the default)
    s : slice
        A slice representation of the bounding box. Returned when ``as_slice``
        is true
    """
    if not img.shape:
        return np.array([], dtype=np.intp)
    r = _bbox.bbox(img)
    if border:
        min1,max1,min2,max2 = r
        min1 = max(0, min1-border)
        min2 = max(0, min2-border)
        max1 += border
        max2 += border
        r = min1,max1,min2,max2
    if as_slice:
        return (slice(r[0],r[1]),slice(r[2],r[3]))
    return r

def croptobbox(img, border=None):
    """
    nimg = croptobbox(img, border=0)

    Returns a version of img cropped to the image's bounding box

    Parameters
    ----------
    img : ndarray
        Integer image array
    border : int, optional
        whether to add a border (default no border)

    Returns
    -------
    nimg : ndarray
        A subimage of img.

    Bugs
    ----
    Note that the border is on the bounding box, not on the final image! This
    means that if the image has a positive pixel on its margin, it will still
    be on the margin.

    This ensures that the result is always a sub-image of the input.
    """
    min1,max1,min2,max2 = bbox(img, border=border)
    return img[min1:max1,min2:max2]

