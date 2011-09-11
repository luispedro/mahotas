# Copyright (C) 2008-2011, Luis Pedro Coelho <luis@luispedro.org>
# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation; either version 2 of the License,
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

from __future__ import division

import _bbox
import numpy as np
from .morph import _verify_is_integer_type

def bbox(img):
    """
    min1,max1,min2,max2 = bbox(img)

    Calculate the bounding box of image img.

    Parameters
    ----------
    img : ndarray
        Any integer image type

    Returns
    -------
    min1,max1,min2,max2 : int,int,int,int
        These are such that ``img[min1:max1, min2:max2]`` contains all non-zero
        pixels
    """
    _verify_is_integer_type(img, 'mahotas.bbox')
    if not img.shape:
        return np.array([], dtype=np.intp)
    return _bbox.bbox(img)

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
    min1,max1,min2,max2 = bbox(img)
    if border:
        min1 = max(0, min1-border)
        min2 = max(0, min2-border)
        max1 += border
        max2 += border
    return img[min1:max1,min2:max2]

