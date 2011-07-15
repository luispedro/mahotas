# Copyright (C) 2008-2010, Luis Pedro Coelho <luis@luispedro.org>
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
import numpy as np

__all__ = ['moments']
def moments(img, p0, p1, cm=None, convert_to_float=True):
    '''
    m = moments(img, p0, p1, cm=(0, 0), convert_to_float=True)

    Returns the p0-p1 moment of image `img`

    The formulat computed is

    \sum_{ij} { img[i,j] (i - c0)**p0 (j - c1)**p1 }

    where cm = (c0,c1). If `cm` is not given, then (0,0) is used.

    If image is of an integer type, then it is internally converted to
    np.float64, unlesss `convert_to_float` is False. The reason is that,
    otherwise, overflow is likely except for small images. Since this
    conversion takes longer than the computation, you can turn it off in case
    where you are sure that your images are small enough for overflow to be an
    issue. Note that no conversion is made if `img` is of any floating point
    type.

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
        whether to convert to floating point

    Returns
    -------
    moment: float
        floating point number

    Bugs
    ----
      It only works for 2-D images
    '''
    if not np.issubdtype(img.dtype, float) and convert_to_float:
        img = img.astype(np.float64)
    r,c = img.shape
    p = np.arange(c)
    if cm is not None:
        p -= cm[1]
    p **= p0
    inter = np.dot(img, p)
    p = np.arange(r)
    if cm is not None:
        p -= cm[0]
    p **= p1
    return np.dot(inter, p)

